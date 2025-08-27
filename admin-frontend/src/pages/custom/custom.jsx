import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import PageHeaderWidgets from '@/components/shared/pageHeader/PageHeaderWidgets';
import Footer from '@/components/shared/Footer';
import CustomTable from '@/components/custom/CustomTable';

const Custom = () => {
    return (
        <>
            <PageHeader>
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <CustomTable title="Inquiries List" />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Custom;
