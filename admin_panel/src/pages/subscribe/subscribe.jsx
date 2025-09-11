import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import PageHeaderWidgets from '@/components/shared/pageHeader/PageHeaderWidgets';
import Footer from '@/components/shared/Footer';
import SubscriberTable from '@/components/subscriber/SubscriberTable';
import { BsDot } from 'react-icons/bs';

const Subscribe = () => {
    return (
        <>
            <PageHeader>
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <SubscriberTable
            title={
              <span className="d-inline-flex align-items-center gap-3">
                <span>Session Details</span>
                <span className="small text-danger d-inline-flex align-items-center">
                  <BsDot className="fs-4 me-1 text-danger" />
                  Rows shown in red have higher number of cheating predictions
                </span>
              </span>
            }
          />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Subscribe;
