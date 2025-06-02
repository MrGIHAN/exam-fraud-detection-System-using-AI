import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import PageHeaderWidgets from '@/components/shared/pageHeader/PageHeaderWidgets';
import Footer from '@/components/shared/Footer';
import PackageTable from '@/components/package/PackageTable';

const Package = () => {
    return (
        <>
            <PageHeader>
                <PageHeaderWidgets addNewLink="/admin/package/create" name="Add New package" />
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <PackageTable title="Location List" />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Package;
